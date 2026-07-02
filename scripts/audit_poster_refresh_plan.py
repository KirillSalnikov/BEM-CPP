#!/usr/bin/env python3

import csv
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


ROOT = Path(__file__).resolve().parents[1]


def opt_value(args: List[str], opt: str) -> Optional[str]:
    try:
        idx = args.index(opt)
    except ValueError:
        return None
    if idx + 1 >= len(args):
        return None
    return args[idx + 1]


def require(cond: bool, msg: str, errors: List[str]) -> None:
    if not cond:
        errors.append(msg)


def main() -> int:
    plan = subprocess.check_output(
        ["bash", "scripts/queue_poster_true_residual_refresh.sh", "--plan"],
        cwd=ROOT,
        universal_newlines=True,
    )
    rows = list(csv.DictReader(plan.splitlines()))
    errors = []  # type: List[str]

    require(len(rows) == 14, f"expected 14 planned cases, got {len(rows)}", errors)
    names = [r["name"] for r in rows]
    require(len(set(names)) == len(names), "case names must be unique", errors)

    for row in rows:
        name = row["name"]
        # queue --plan prints shell-escaped argv with spaces escaped as "\ ".
        # The planned arguments do not contain meaningful spaces in values, so
        # decode that formatting before token-level checks.
        args = shlex.split(row["args"].replace("\\ ", " "))
        shape = opt_value(args, "--shape")
        obj = opt_value(args, "--obj")
        quad = opt_value(args, "--quad")
        digits = opt_value(args, "--fmm-digits")
        tol = opt_value(args, "--gmres-tol")
        restart = opt_value(args, "--gmres-restart")

        if name.startswith("sphere_"):
            require(shape == "sphere", f"{name}: sphere case must use --shape sphere", errors)
        if name.startswith("hex_"):
            require(shape == "hex_prism", f"{name}: hex case must use --shape hex_prism", errors)
        if name.startswith("dust_"):
            require(obj is not None, f"{name}: dust case must use --obj", errors)
            if obj is not None:
                require((ROOT / obj).exists(), f"{name}: missing OBJ {obj}", errors)
            require("--accurate" in args, f"{name}: dust production case must use --accurate", errors)

        require(digits is not None and int(digits) >= 7,
                f"{name}: poster refresh requires --fmm-digits >= 7", errors)
        require(tol is not None and float(tol) <= 1e-5,
                f"{name}: poster refresh requires --gmres-tol <= 1e-5", errors)
        require(opt_value(args, "--gmres-max-cycles") is not None and
                int(opt_value(args, "--gmres-max-cycles") or "0") >= 80,
                f"{name}: poster refresh requires --gmres-max-cycles >= 80", errors)

        m = re.search(r"_q(\d+)_d(\d+)_tol(\d+)e(\d+)$", name)
        if m:
            q_name = int(m.group(1))
            d_name = int(m.group(2))
            tol_name = float(m.group(3)) * (10.0 ** (-int(m.group(4))))
            require(quad is not None and int(quad) == q_name,
                    f"{name}: case q{q_name} != --quad {quad}", errors)
            require(digits is not None and int(digits) == d_name,
                    f"{name}: case d{d_name} != --fmm-digits {digits}", errors)
            require(tol is not None and float(tol) <= tol_name,
                    f"{name}: --gmres-tol {tol} exceeds name tolerance {tol_name:g}", errors)

        if name.startswith("dust_"):
            require(restart is not None and int(restart) >= 1000,
                    f"{name}: dust requires --gmres-restart >= 1000", errors)

    if errors:
        for e in errors:
            print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"poster refresh plan ok: {len(rows)} cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
