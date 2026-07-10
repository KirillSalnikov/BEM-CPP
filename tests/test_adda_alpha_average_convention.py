#!/usr/bin/env python3
"""BEM fast-alpha averaging must keep ADDA's orientation convention."""

from pathlib import Path
from typing import List


ROOT = Path(__file__).resolve().parents[1]


def require_all(text: str, snippets: List[str], name: str) -> None:
    missing = [snippet for snippet in snippets if snippet not in text]
    assert not missing, f"{name} is missing ADDA-compatible alpha snippets: {missing}"


def main() -> int:
    main_cpp = (ROOT / "src" / "main.cpp").read_text()
    farfield_cu = (ROOT / "src" / "farfield.cu").read_text()
    orient_cpp = (ROOT / "src" / "orient.cpp").read_text()

    # ADDA CalculateE.c, yzplane orientation average:
    # s2 =  co*s20 + si*s30; s3 = -si*s20 + co*s30
    # s4 =  co*s40 + si*s10; s1 = -si*s40 + co*s10
    yz_mix = [
        "J0[i] = ca * jp + sa * ju;",
        "M0[i] = ca * mp + sa * mu;",
        "J1[i] = -sa * jp + ca * ju;",
        "M1[i] = -sa * mp + ca * mu;",
    ]
    require_all(main_cpp, yz_mix, "main.cpp host alpha mixing")

    yz_mix_cuda = [
        "cJ0_re = ca * jp_re + sa * ju_re;",
        "cM0_re = ca * mp_re + sa * mu_re;",
        "cJ1_re = -sa * jp_re + ca * ju_re;",
        "cM1_re = -sa * mp_re + ca * mu_re;",
    ]
    require_all(farfield_cu, yz_mix_cuda, "farfield.cu CUDA alpha mixing")

    rotate_minus_alpha = [
        "ca * v.x + sa * v.y",
        "-sa * v.x + ca * v.y",
        "double x = ca * rh0[0] + sa * rh0[1];",
        "double y = -sa * rh0[0] + ca * rh0[1];",
    ]
    require_all(main_cpp + farfield_cu, rotate_minus_alpha, "alpha geometry rotation")

    require_all(
        orient_cpp,
        [
            "R = Rz(alpha) * Ry(beta) * Rz(gamma)",
            "o.RT = R.T();",
        ],
        "Euler orientation",
    )

    print("adda alpha average convention: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
