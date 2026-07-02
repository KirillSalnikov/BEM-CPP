#!/usr/bin/env python3

from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import poster_a0.validate_poster as validate_poster  # noqa: E402


def main() -> int:
    true_values = ["True", "true", "1", "yes", "PASS", True]
    false_values = ["False", "false", "0", "no", "", False, None]
    for value in true_values:
        assert validate_poster.csv_true(value), value
    for value in false_values:
        assert not validate_poster.csv_true(value), value

    source = (ROOT / "poster_a0" / "validate_poster.py").read_text()
    literal_bool_checks = re.findall(r"(?:==|!=)\s*[\"'](?:True|False)[\"']", source)
    assert not literal_bool_checks, literal_bool_checks

    print("validate poster truth flags: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
