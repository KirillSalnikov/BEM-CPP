#!/usr/bin/env python3
"""M2L setup must not recompute radial Hankel factors per angle."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    source = (ROOT / "src" / "fmm.cu").read_text()
    begin = source.index("std::vector<cldouble> radial_coeff")
    angular = source.index("std::vector<cdouble> T(L)", begin)
    end = source.index("for (int ll = 0; ll < L; ll++)", angular)
    radial_setup = source[begin:angular]
    angular_setup = source[end:source.index("transfer_cache.push_back", end)]

    assert "spherical_hankel1_extended(l, kd)" in radial_setup
    assert "spherical_hankel1_extended" not in angular_setup
    assert "sum += radial_coeff[l] * P_next" in angular_setup
    print("FMM M2L radial cache: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
