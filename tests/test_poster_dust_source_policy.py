#!/usr/bin/env python3
"""Guard primary poster dust plots against stale fast BEM sources."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from poster_a0.make_assets import PRODUCTION_PAIR_KEYS, is_primary_dust_bem_path  # noqa: E402
from poster_a0.validate_poster import is_primary_dust_bem_source  # noqa: E402


FORBIDDEN_PRIMARY_DUST_TOKENS = (
    "dust_tol_sweep",
    "current_time_refresh_v2/dust",
    "current_time_refresh_v3/mem/dust",
    "current_time_refresh_v4/mem/dust",
    "dust_ka20_current_attempt",
    "dust_ka30_current_attempt",
    "tol2em2",
    "complexop",
)

ALLOWED_PRIMARY_DUST_TOKENS = (
    "balanced_q7_d6_tol5e4",
    "balanced_q9_d6_tol5e4",
    "pmchwt_q7_d5_tol1e3",
    "muller2b",
)


def between(text: str, start: str, end: str) -> str:
    i = text.index(start)
    j = text.index(end, i)
    return text[i:j]


def assert_clean_primary_block(name: str, block: str) -> None:
    stale = [token for token in FORBIDDEN_PRIMARY_DUST_TOKENS if token in block]
    assert not stale, f"{name} contains stale dust sources: {stale}"
    assert any(token in block for token in ALLOWED_PRIMARY_DUST_TOKENS), name


def main() -> int:
    assert is_primary_dust_bem_path("runs/x/dust_ka10_balanced_q7_d6_tol5e4.json")
    assert is_primary_dust_bem_source("runs/x/dust_ka10_balanced_q9_d6_tol5e4.json")
    assert is_primary_dust_bem_path("runs/x/dust_ka10_gmsh3400_muller2b_n181.json")
    assert is_primary_dust_bem_source("runs/x/dust_ka5_gmsh3400_pmchwt_q7_d5_tol1e3.json")
    assert not is_primary_dust_bem_source("runs/x/dust_ka5_f800_pmchwt.json")
    assert ("пылевая частица", 2.0) not in PRODUCTION_PAIR_KEYS
    assert ("пылевая частица", 5.0) in PRODUCTION_PAIR_KEYS

    text = (ROOT / "poster_a0" / "make_assets.py").read_text()
    shape_time_block = between(
        text,
        '"пылевая частица": [\n            (5.0, 5200,',
        "    rows = []\n    def bem_backend_label",
    )
    assert_clean_primary_block("shape-time dust specs", shape_time_block)

    accuracy_block = between(
        text,
        '("пылевая частица", 5.0, 5200,',
        "    dust30_mueller =",
    )
    assert_clean_primary_block("ADDA-OCL accuracy dust specs", accuracy_block)

    vram_block = between(
        text,
        '("пылевая частица", "пыль", 5.0, 0,',
        "    production_rows = []",
    )
    assert_clean_primary_block("production VRAM dust specs", vram_block)

    overrides_block = between(
        text,
        '"dust_ka10_gmsh3400_balanced_q7_d6_tol5e4": (',
        "    def mem_peak_gb",
    )
    assert_clean_primary_block("VRAM override dust specs", overrides_block)

    validate_text = (ROOT / "poster_a0" / "validate_poster.py").read_text()
    primary_pair_block = between(
        validate_text,
        "required_pairs = {",
        "    row_keys =",
    )
    assert '("пылевая частица", 2.0)' not in primary_pair_block
    assert '("пылевая частица", 5.709)' not in primary_pair_block
    assert '("пылевая частица", 5.0)' in primary_pair_block

    print("poster dust source policy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
