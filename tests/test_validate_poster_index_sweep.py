#!/usr/bin/env python3
"""Poster validator must reject legacy M11-only index sweep tables."""

import tempfile
from pathlib import Path
import sys
from contextlib import redirect_stdout
from io import StringIO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import poster_a0.validate_poster as validate_poster


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        assets = Path(tmp)
        (assets / "table_index_sweep.csv").write_text(
            "ka,n,ref,status,shape_l2\n"
            "5,1.5,3,PASS,0.01\n"
        )
        old_assets = validate_poster.ASSETS
        validate_poster.ASSETS = assets
        try:
            captured = StringIO()
            with redirect_stdout(captured):
                try:
                    validate_poster.check_index_sweep_full_mueller()
                except SystemExit as exc:
                    assert exc.code == 1, exc
                else:
                    raise AssertionError("legacy index sweep table was accepted")
            assert "lacks full-Mueller columns" in captured.getvalue()
        finally:
            validate_poster.ASSETS = old_assets

        (assets / "table_index_sweep.csv").write_text(
            "ka,n,ref,status,shape_l2,pass10_shape_l2,pass10_full_mueller,"
            "max_main_floor2,max16_floor2,worst_component,worst_component_error,"
            "failed_main_10pct,failed_all_20pct\n"
            "5,1.5,3,FAIL,0.01,True,False,0.01,0.3,M34,0.3,,M34\n"
        )
        (assets / "table_index_sweep_selected.csv").write_text(
            "ka,n,ref,status,shape_l2,pass10_shape_l2,pass10_full_mueller,"
            "max_main_floor2,max16_floor2,worst_component,worst_component_error,"
            "failed_main_10pct,failed_all_20pct\n"
        )
        (assets / "table_index_sweep_coverage.csv").write_text(
            "ka,n,any_pass,status\n"
            "5,1.5,False,PENDING\n"
        )
        old_assets = validate_poster.ASSETS
        validate_poster.ASSETS = assets
        try:
            validate_poster.check_index_sweep_full_mueller()
        finally:
            validate_poster.ASSETS = old_assets

        (assets / "table_index_sweep_coverage.csv").write_text(
            "ka,n,any_pass,status\n"
            "5,1.5,True,PASS\n"
        )
        old_assets = validate_poster.ASSETS
        validate_poster.ASSETS = assets
        try:
            captured = StringIO()
            with redirect_stdout(captured):
                try:
                    validate_poster.check_index_sweep_full_mueller()
                except SystemExit as exc:
                    assert exc.code == 1, exc
                else:
                    raise AssertionError("coverage PASS without full Mueller source was accepted")
            assert "coverage PASS lacks full-Mueller source" in captured.getvalue()
        finally:
            validate_poster.ASSETS = old_assets

    print("validate poster index sweep: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
