#!/usr/bin/env python3
"""SCUFF-style structural regression tests for analytic sphere Mueller data."""

import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from verify_mie import mie_m11, mie_mueller  # noqa: E402


ZERO_COMPONENTS = (
    (0, 2), (0, 3),
    (1, 2), (1, 3),
    (2, 0), (2, 1),
    (3, 0), (3, 1),
)


def rel_close(a: float, b: float, tol: float = 1e-10) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))


def main() -> int:
    theta = [float(i) for i in range(181)]
    # Include a weakly absorbing sphere because this is where sign/convention
    # mistakes tend to hide behind a plausible M11 curve.
    mu = mie_mueller(theta, complex(1.6, 0.002), 12.0)
    m11 = mie_m11(theta, complex(1.6, 0.002), 12.0)

    worst_identity = 0.0
    worst_bound = 0.0
    for t, _angle in enumerate(theta):
        assert rel_close(mu[0][0][t], m11[t]), ("M11 wrapper mismatch", t)
        assert mu[0][0][t] >= 0.0, ("negative M11", t, mu[0][0][t])
        assert rel_close(mu[0][0][t], mu[1][1][t]), ("M11/M22", t)
        assert rel_close(mu[0][1][t], mu[1][0][t]), ("M12/M21", t)
        assert rel_close(mu[2][2][t], mu[3][3][t]), ("M33/M44", t)
        assert rel_close(mu[2][3][t], -mu[3][2][t]), ("M34/M43", t)
        for i, j in ZERO_COMPONENTS:
            assert rel_close(mu[i][j][t], 0.0), (f"M{i + 1}{j + 1} should be zero", t, mu[i][j][t])

        scale = max(mu[0][0][t], 1e-300)
        for i in range(4):
            for j in range(4):
                worst_bound = max(worst_bound, abs(mu[i][j][t]) / scale)
                assert abs(mu[i][j][t]) <= scale * (1.0 + 1e-10), (
                    f"|M{i + 1}{j + 1}| > M11", t, mu[i][j][t], scale
                )

        lhs = mu[0][0][t] ** 2 - mu[0][1][t] ** 2
        rhs = mu[2][2][t] ** 2 + mu[2][3][t] ** 2
        denom = max(1.0, abs(lhs), abs(rhs))
        worst_identity = max(worst_identity, abs(lhs - rhs) / denom)
        assert rel_close(lhs, rhs, 1e-10), ("Mueller amplitude invariant", t, lhs, rhs)

    # Forward and backward scattering for a sphere have no circular block
    # rotation in this amplitude convention.
    for t in (0, 180):
        assert abs(mu[2][3][t]) <= 1e-9 * max(1.0, abs(mu[0][0][t])), ("endpoint M34", t, mu[2][3][t])

    assert math.isfinite(worst_identity)
    assert worst_bound <= 1.0 + 1e-10
    print("mie mueller symmetry: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
