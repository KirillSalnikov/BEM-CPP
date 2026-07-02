#!/usr/bin/env python3
"""Checks for full sphere Mueller matrix from Mie amplitudes."""

import math
import json
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verify_mie import compare, mie_m11, mie_mueller  # noqa: E402


def assert_close(a: float, b: float, tol: float = 1e-12) -> None:
    scale = max(1.0, abs(a), abs(b))
    assert abs(a - b) <= tol * scale, (a, b)


def main() -> int:
    theta = [0.0, 30.0, 90.0, 150.0, 180.0]
    m = complex(1.3116, 0.0)
    ka = 5.0
    m11 = mie_m11(theta, m, ka)
    mu = mie_mueller(theta, m, ka)

    assert len(mu) == 4
    assert len(mu[0]) == 4
    assert len(mu[0][0]) == len(theta)

    for t in range(len(theta)):
        assert_close(mu[0][0][t], m11[t])
        assert_close(mu[1][1][t], mu[0][0][t])
        assert_close(mu[1][0][t], mu[0][1][t])
        assert_close(mu[3][3][t], mu[2][2][t])
        assert_close(mu[3][2][t], -mu[2][3][t])
        assert mu[0][0][t] >= 0.0
        for i, j in ((0, 2), (0, 3), (1, 2), (1, 3),
                     (2, 0), (2, 1), (3, 0), (3, 1)):
            assert math.isfinite(mu[i][j][t])
            assert_close(mu[i][j][t], 0.0)

    distorted = json.loads(json.dumps(mu))
    norm = max(abs(distorted[0][0][0]), 1.0)
    for t in range(len(theta)):
        distorted[2][3][t] += 0.3 * norm
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "distorted.json"
        path.write_text(json.dumps({"theta": theta, "mueller": distorted}))
        with redirect_stdout(StringIO()):
            summary = compare(path, m.real, m.imag, ka)
    assert summary["worst_component"] == "M34", summary
    assert "M34" in summary["failed_all_20pct"], summary

    print("verify mie mueller: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
