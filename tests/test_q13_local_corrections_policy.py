#!/usr/bin/env python3
"""Guard q13 edge-aware local corrections for OBJ/FMM accuracy."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    src = (ROOT / "src" / "bem_fmm.cu").read_text()
    block_start = src.index("const bool edge_delta_enabled")
    block_end = src.index("const bool local_delta_enabled", block_start)
    policy = src[block_start:block_end]
    if "quad_order < 13" in policy:
        raise AssertionError(
            "local/edge correction policy must not disable corrections at q13"
        )
    if 'bem_env_flag_enabled("BEM_AUTO_LOCAL_CORRECTIONS", true)' not in policy:
        raise AssertionError("auto local corrections default must stay enabled")
    print("q13 local correction policy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
