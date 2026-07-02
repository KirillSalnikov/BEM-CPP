#!/usr/bin/env python3

from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import poster_a0.validate_poster as validate_poster  # noqa: E402


def expect_fail(row, message):
    captured = StringIO()
    with redirect_stdout(captured):
        try:
            validate_poster.check_production_vram_accuracy_row(row, "сфера", 5.0)
        except SystemExit as exc:
            assert exc.code == 1, exc
        else:
            raise AssertionError("invalid production VRAM accuracy row was accepted")
    assert message in captured.getvalue(), captured.getvalue()


def main() -> int:
    validate_poster.check_production_vram_accuracy_row(
        {"status": "PASS", "pass10": "True", "accuracy_source": "table_accuracy_matrix_15.csv"},
        "сфера",
        5.0,
    )
    validate_poster.check_production_vram_accuracy_row(
        {"status": "missing_accuracy", "pass10": "False", "accuracy_source": ""},
        "сфера",
        5.0,
    )
    expect_fail(
        {"status": "PASS", "pass10": "True", "accuracy_source": ""},
        "pass10 without accuracy source",
    )
    expect_fail(
        {"status": "missing_accuracy", "pass10": "True", "accuracy_source": "table_accuracy_matrix_15.csv"},
        "pass10 without PASS status",
    )

    print("validate poster VRAM accuracy: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
